"""Profile-free SAM3 video annotation adapter for RobotWin datasets.

The control flow intentionally follows ``annotate/video_annotate.py``:

* load all episode frames;
* split long videos into chunks;
* add one text prompt per object and propagate it through the chunk;
* merge the per-object outputs by frame;
* save COCO annotations and a rendered video.

Unlike the scenario pipeline, this module has no dependency on ``PROFILES`` or
the fixed object table in ``annotate/config/profile.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from annotate.utils.coco_io import save_outputs_merged_to_coco_json
from sam3.model_builder import build_sam3_video_predictor


VIDEO_CHUNK_SIZE = 1000
COLORS_BGR = (
    (0, 255, 0),      # obj_id 1: 绿色
    (0, 0, 255),      # obj_id 2: 红色
    (255, 255, 0),    # obj_id 3: 青色
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
)


@dataclass(frozen=True)
class ObjectSpec:
    """One explicitly configured annotation object."""

    obj_id: int
    name: str
    prompt: str


def parse_object_specs(raw_objects: list[list[str]]) -> list[ObjectSpec]:
    """Parse repeated CLI entries in the same spirit as profile.py's OBJ table."""
    objects = [
        ObjectSpec(obj_id=int(obj_id), name=name, prompt=prompt)
        for obj_id, name, prompt in raw_objects
    ]
    ids = [obj.obj_id for obj in objects]
    names = [obj.name for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Object IDs must be unique: {ids}")
    if len(names) != len(set(names)):
        raise ValueError(f"Object names must be unique: {names}")
    if any(obj.obj_id <= 0 for obj in objects):
        raise ValueError(f"Object IDs must be positive: {ids}")
    if any(not obj.prompt.strip() for obj in objects):
        raise ValueError("Object prompts must not be empty")
    return objects


def normalize_camera(camera: str) -> str:
    value = camera.strip()
    if value.startswith("observation.images."):
        value = value[len("observation.images.") :]
    if value.startswith("cam_"):
        value = "cam" + value[4:]
    if not value.startswith("cam") or not value[3:].isdigit():
        raise ValueError(
            f"Invalid camera {camera!r}; use cam2, cam3, cam4, or "
            "observation.images.cam2."
        )
    return value


def load_video_frames(video_path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    return frames


class RobotWinVideoPromptRunner:
    """Run several independent text prompts on one video chunk."""

    KEYS = ("out_obj_ids", "out_track_ids", "out_boxes_xywh", "out_binary_masks")

    def __init__(self, predictor: Any, objects: list[ObjectSpec]):
        self.predictor = predictor
        self.objects = objects

    @staticmethod
    def relabel_output(output: dict[str, Any], obj_id: int) -> dict[str, Any]:
        """Give every mask from this prompt its stable prompt/category ID."""
        result = dict(output)
        masks = result.get("out_binary_masks")
        old_ids = result.get("out_obj_ids")
        if isinstance(masks, np.ndarray) and masks.ndim >= 1:
            count = masks.shape[0]
        elif isinstance(old_ids, np.ndarray):
            count = old_ids.shape[0]
        else:
            count = 0
        dtype = old_ids.dtype if isinstance(old_ids, np.ndarray) else np.int64
        result["out_obj_ids"] = np.full((count,), obj_id, dtype=dtype)
        return result

    def propagate_in_video(
        self, session_id: str, obj_id: int
    ) -> dict[int, dict[str, Any]]:
        outputs: dict[int, dict[str, Any]] = {}
        for response in self.predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
            }
        ):
            outputs[response["frame_index"]] = self.relabel_output(
                response["outputs"], obj_id
            )
        return outputs

    def add_prompt_and_propagate(
        self,
        frames: list[Image.Image],
        prompt: str,
        obj_id: int,
        bounding_box: np.ndarray | None = None,
    ) -> dict[int, dict[str, Any]]:
        response = self.predictor.handle_request(
            request={"type": "start_session", "resource_path": frames}
        )
        session_id = response["session_id"]
        try:
            request: dict[str, Any] = {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
                "obj_id": obj_id,
            }
            if bounding_box is not None:
                request["bounding_boxes"] = [bounding_box]
                request["bounding_box_labels"] = [1]

            response = self.predictor.handle_request(request=request)
            outputs = self.propagate_in_video(session_id, obj_id)
            # Propagation normally contains frame 0, but retain the direct
            # prompt result for predictors/configurations that do not yield it.
            outputs.setdefault(
                int(response.get("frame_index", 0)),
                self.relabel_output(response["outputs"], obj_id),
            )
            return outputs
        finally:
            self.predictor.handle_request(
                request={"type": "close_session", "session_id": session_id}
            )

    def merge_outputs_per_frame(
        self, outputs_list: list[dict[int, dict[str, Any]]]
    ) -> dict[int, dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        for outputs in outputs_list:
            for frame_idx, output in outputs.items():
                obj_ids = output.get("out_obj_ids")
                masks = output.get("out_binary_masks")
                if not isinstance(obj_ids, np.ndarray) or not isinstance(
                    masks, np.ndarray
                ):
                    continue
                if obj_ids.shape[0] != masks.shape[0]:
                    continue

                if frame_idx not in merged:
                    merged[frame_idx] = {
                        key: output[key]
                        for key in self.KEYS
                        if key in output and isinstance(output[key], np.ndarray)
                    }
                    continue

                for key in self.KEYS:
                    if key in output and isinstance(output[key], np.ndarray):
                        merged[frame_idx][key] = np.concatenate(
                            [merged[frame_idx][key], output[key]], axis=0
                        )
        return merged

    def run_prompts_and_merge(
        self,
        frames: list[Image.Image],
        bounding_boxes: dict[int, np.ndarray] | None = None,
    ) -> dict[int, dict[str, Any]]:
        outputs_list = []
        for obj in self.objects:
            box = bounding_boxes.get(obj.obj_id) if bounding_boxes else None
            print(
                f"[PROMPT] obj_id={obj.obj_id} name={obj.name!r} "
                f"prompt={obj.prompt!r} box={box is not None}"
            )
            outputs_list.append(
                self.add_prompt_and_propagate(
                    frames, obj.prompt, obj.obj_id, box
                )
            )
        return self.merge_outputs_per_frame(outputs_list)

    def run_in_chunks_and_merge(
        self, frames: list[Image.Image]
    ) -> dict[int, dict[str, Any]]:
        outputs_all: dict[int, dict[str, Any]] = {}
        for start in range(0, len(frames), VIDEO_CHUNK_SIZE):
            end = min(start + VIDEO_CHUNK_SIZE, len(frames))
            chunk_frames = frames[start:end]
            boxes = None
            if start > 0 and start - 1 in outputs_all:
                previous = outputs_all[start - 1]
                boxes = {
                    int(obj_id): box
                    for obj_id, box in zip(
                        previous.get("out_obj_ids", []),
                        previous.get("out_boxes_xywh", []),
                    )
                }
            print(
                f"[CHUNK] start={start} end={end} "
                f"frames={len(chunk_frames)} objects={len(self.objects)}"
            )
            chunk_outputs = self.run_prompts_and_merge(chunk_frames, boxes)
            for local_idx, output in chunk_outputs.items():
                outputs_all[start + local_idx] = output
        return outputs_all


def render_video(
    video_path: Path,
    output_path: Path,
    outputs: dict[int, dict[str, Any]],
) -> None:
    frames = load_video_frames(video_path)
    height, width = frames[0].height, frames[0].width
    fps = cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS) or 50.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output_path}")

    try:
        for frame_idx, image in enumerate(frames):
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            output = outputs.get(frame_idx)
            if output is not None:
                masks = output.get("out_binary_masks")
                ids = output.get("out_obj_ids")
                if isinstance(masks, np.ndarray) and isinstance(ids, np.ndarray):
                    for mask, obj_id in zip(masks, ids):
                        if mask is None or mask.size == 0 or not mask.any():
                            continue
                        color = COLORS_BGR[(int(obj_id) - 1) % len(COLORS_BGR)]
                        mask_u8 = mask.astype(np.uint8) * 255
                        overlay = frame.copy()
                        overlay[mask_u8 > 0] = color
                        frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
                        contours, _ = cv2.findContours(
                            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(frame, contours, -1, color, 2)
            writer.write(frame)
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate RobotWin episode videos with multiple SAM3 prompts."
    )
    parser.add_argument(
        "--data-path", type=Path, required=True, help="RobotWin dataset directory."
    )
    parser.add_argument(
        "--camera",
        required=True,
        help="Camera, for example cam2 or observation.images.cam2.",
    )
    parser.add_argument(
        "--object",
        action="append",
        nargs=3,
        required=True,
        metavar=("ID", "NAME", "PROMPT"),
        help=(
            "Annotation object as explicit ID, name, and prompt. Repeat this "
            "option for multiple objects; e.g. --object 1 box 'playing card box'."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Default: <data-path>_sam3_<camera>.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        type=int,
        default=None,
        help=(
            "Only process the specified episode index; repeat for multiple episodes. "
            "If omitted, process all episodes."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes whose COCO output already exists; default is overwrite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    objects = parse_object_specs(args.object)
    data_path = args.data_path.expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")
    camera = normalize_camera(args.camera)
    camera_dir = f"observation.images.{camera}"
    source_video_dir = data_path / "videos" / "chunk-000" / camera_dir
    if not source_video_dir.is_dir():
        raise FileNotFoundError(f"Camera directory does not exist: {source_video_dir}")

    output_path = args.output_path
    if output_path is None:
        output_path = data_path.parent / f"{data_path.name}_sam3_{camera}"
    output_path = output_path.expanduser().resolve()
    annotation_dir = output_path / "videos_annotate" / "chunk-000" / camera_dir
    render_dir = output_path / "videos_render" / "chunk-000" / camera_dir
    annotation_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(source_video_dir.glob("episode_*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No episode_*.mp4 files found in {source_video_dir}")
    if args.episode is not None:
        selected = {
            source_video_dir / f"episode_{episode:06d}.mp4"
            for episode in args.episode
        }
        missing = sorted(path.name for path in selected if not path.exists())
        if missing:
            raise FileNotFoundError(
                f"Requested episode video(s) do not exist in {source_video_dir}: "
                f"{', '.join(missing)}"
            )
        videos = [video for video in videos if video in selected]

    config_path = output_path / "meta" / "sam3_annotation_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "source_data_path": str(data_path),
                "camera": camera,
                "objects": [
                    {"id": obj.obj_id, "name": obj.name, "prompt": obj.prompt}
                    for obj in objects
                ],
                "episodes": args.episode,
                "video_count": len(videos),
            },
            indent=2,
        )
        + "\n"
    )

    predictor = build_sam3_video_predictor()
    runner = RobotWinVideoPromptRunner(predictor, objects)
    for index, video_path in enumerate(videos, start=1):
        coco_path = annotation_dir / f"{video_path.stem}.coco.json"
        render_path = render_dir / video_path.name
        if coco_path.exists() and args.skip_existing:
            print(f"[{index}/{len(videos)}] Skip existing: {coco_path}")
            continue
        print(f"[{index}/{len(videos)}] Processing {video_path.name}")
        frames = load_video_frames(video_path)
        outputs = runner.run_in_chunks_and_merge(frames)
        save_outputs_merged_to_coco_json(
            outputs,
            coco_path,
            video_name=video_path.stem,
            category_names_by_id={obj.obj_id: obj.name for obj in objects},
        )
        render_video(video_path, render_path, outputs)

    print(f"[DONE] Output: {output_path}")


if __name__ == "__main__":
    main()
