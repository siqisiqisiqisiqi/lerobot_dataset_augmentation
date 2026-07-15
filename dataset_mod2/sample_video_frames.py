from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_SCENARIOS = ("scenario_8",)
DEFAULT_CAMERA_NAME = "observation.images.cam2"
DEFAULT_NUM_FRAMES = 10
DEFAULT_VIDEO_NAME = "episode_000049.mp4"
DEFAULT_VIDEO_INDEX = 10


def iter_date_dirs(
    root: Path,
    scenario_names: tuple[str, ...],
    dates: set[str] | None,
    camera_name: str,
) -> list[tuple[str, Path]]:
    date_dirs: list[tuple[str, Path]] = []

    for scenario_name in scenario_names:
        scenario_dir = root / scenario_name
        if not scenario_dir.exists():
            continue

        for date_dir in sorted(path for path in scenario_dir.iterdir() if path.is_dir()):
            if dates is not None and date_dir.name not in dates:
                continue

            camera_dir = date_dir / "videos" / "chunk-000" / camera_name
            if camera_dir.exists():
                date_dirs.append((scenario_name, date_dir))

    return date_dirs


def select_video(
    date_dir: Path,
    camera_name: str,
    video_name: str | None,
    video_index: int,
) -> Path:
    camera_dir = date_dir / "videos" / "chunk-000" / camera_name

    if video_name is not None:
        video_path = camera_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Could not find {video_path}")
        return video_path

    videos = sorted(camera_dir.glob("episode_*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No episode_*.mp4 files found in {camera_dir}")
    if video_index < 0 or video_index >= len(videos):
        raise IndexError(
            f"Video index {video_index} is out of range for {camera_dir}; "
            f"found {len(videos)} videos."
        )

    return videos[video_index]


def sample_indices(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Video has no readable frames")
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than 0")
    if num_frames == 1:
        return [frame_count // 2]

    last_index = frame_count - 1
    return [
        round(index * last_index / (num_frames - 1))
        for index in range(num_frames)
    ]


def sample_video(
    video_path: Path,
    output_dir: Path,
    num_frames: int,
    overwrite: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    indices = sample_indices(frame_count, num_frames)
    saved_frames = []

    try:
        for sample_number, frame_index in enumerate(indices):
            output_path = output_dir / f"{video_path.stem}_frame_{sample_number:03d}.jpg"
            if output_path.exists() and not overwrite:
                saved_frames.append(
                    {
                        "sample_number": sample_number,
                        "frame_index": frame_index,
                        "image": str(output_path),
                    }
                )
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read frame {frame_index} from {video_path}"
                )

            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Could not write image: {output_path}")

            saved_frames.append(
                {
                    "sample_number": sample_number,
                    "frame_index": frame_index,
                    "image": str(output_path),
                }
            )
    finally:
        capture.release()

    return {
        "source_video": str(video_path),
        "frame_count": frame_count,
        "fps": fps,
        "sampled_frames": saved_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select one camera video from each scenario/date folder and sample frames "
            "into scenario_x_sampled_image/date."
        )
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
        help="Scenario folders to process.",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Optional date folder names to process, such as 2025.12.01 2026.01.15.",
    )
    parser.add_argument(
        "--camera-name",
        default=DEFAULT_CAMERA_NAME,
        help="Camera folder under videos/chunk-000 to sample from.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of evenly spaced frames to sample from each selected video.",
    )
    parser.add_argument(
        "--video-name",
        default=DEFAULT_VIDEO_NAME,
        help=(
            "Optional exact video filename to use for every date. "
            "Pass an empty value to select by --video-index."
        ),
    )
    parser.add_argument(
        "--video-index",
        type=int,
        default=DEFAULT_VIDEO_INDEX,
        help="Sorted episode_*.mp4 index to use when --video-name is empty.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing sampled frame images.",
    )
    args = parser.parse_args()

    dates = set(args.dates) if args.dates else None
    date_dirs = iter_date_dirs(
        args.root,
        tuple(args.scenarios),
        dates,
        args.camera_name,
    )
    video_name = args.video_name or None

    for scenario_name, date_dir in date_dirs:
        video_path = select_video(
            date_dir,
            args.camera_name,
            video_name,
            args.video_index,
        )
        output_dir = args.root / f"{scenario_name}_sampled_image" / date_dir.name
        metadata = sample_video(
            video_path=video_path,
            output_dir=output_dir,
            num_frames=args.num_frames,
            overwrite=args.overwrite,
        )

        metadata_path = output_dir / "sampled_frames.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")

        print(f"Sampled {video_path} -> {output_dir}")

    print(f"Completed {len(date_dirs)} date folders.")


if __name__ == "__main__":
    main()
