#!/usr/bin/env python3
import argparse
import glob
import re
import shutil
from pathlib import Path

EP_RE = re.compile(r"episode_(\d+)")

def get_max_index(files):
    if not files:
        raise RuntimeError("No mp4 files found.")
    m = EP_RE.search(Path(files[-1]).name)
    if not m:
        raise RuntimeError(f"Bad filename: {Path(files[-1]).name}")
    return int(m.group(1))

def copy_with_offset(src_root: Path, dst_root: Path, offset: int):
    dst_root.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(src_root / "*.mp4")))
    if not files:
        raise RuntimeError(f"No mp4 files in {src_root}")
    for f in files:
        m = EP_RE.search(Path(f).name)
        if not m:
            raise RuntimeError(f"Bad filename: {Path(f).name}")
        idx = int(m.group(1))
        new_file = dst_root / f"episode_{idx + offset:06d}.mp4"
        shutil.copy(f, str(new_file))

def duplicate_in_place(folder: Path, offset: int):
    files = sorted(glob.glob(str(folder / "*.mp4")))
    if not files:
        raise RuntimeError(f"No mp4 files in {folder}")
    for f in files:
        m = EP_RE.search(Path(f).name)
        if not m:
            raise RuntimeError(f"Bad filename: {Path(f).name}")
        idx = int(m.group(1))
        new_file = folder / f"episode_{idx + offset:06d}.mp4"
        shutil.copy(f, str(new_file))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam2-src", default=None)
    parser.add_argument("--cam3-src", default=None)
    parser.add_argument("--cam4-src", default=None)
    parser.add_argument("--dest-dir", required=True, help="e.g. ./data/scenario_1")
    args = parser.parse_args()

    cam2_src = Path(args.cam2_src) if args.cam2_src else None
    cam3_src = Path(args.cam3_src) if args.cam3_src else None
    cam4_src = Path(args.cam4_src) if args.cam4_src else None

    if cam2_src is None and cam3_src is None and cam4_src is None:
        print("No augmented video source provided. Nothing to do.")
        return

    dest_dir = Path(args.dest_dir)

    # Use existing cam1 as the base to avoid index collision
    dest_cam1 = dest_dir / "videos/chunk-000/observation.images.cam1"
    cam1_files = sorted(glob.glob(str(dest_cam1 / "*.mp4")))
    max_index = get_max_index(cam1_files)
    offset = max_index + 1

    if cam2_src is not None:
        cam2_dst = dest_dir / "videos/chunk-000/observation.images.cam2"
        copy_with_offset(cam2_src, cam2_dst, offset)
        print("Complete the camera 2 augmentation.")

    if cam3_src is not None:
        cam3_dst = dest_dir / "videos/chunk-000/observation.images.cam3"
        copy_with_offset(cam3_src, cam3_dst, offset)
        print("Complete the camera 3 augmentation.")

    if cam4_src is not None:
        cam4_dst = dest_dir / "videos/chunk-000/observation.images.cam4"
        copy_with_offset(cam4_src, cam4_dst, offset)
        print("Complete the camera 4 augmentation.")

    # Duplicate existing videos for cameras that didn't get new sources
    duplicate_camera_list = ["observation.images.cam1"]
    if cam2_src is None:
        duplicate_camera_list.append("observation.images.cam2")
    if cam3_src is None:
        duplicate_camera_list.append("observation.images.cam3")
    if cam4_src is None:
        duplicate_camera_list.append("observation.images.cam4")

    for cam in duplicate_camera_list:
        folder = dest_dir / f"videos/chunk-000/{cam}"
        duplicate_in_place(folder, offset)
        print(f"Complete the {cam.split('.')[-1]} augmentation.")

    print("Done")

if __name__ == "__main__":
    main()