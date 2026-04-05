import os
import json
import argparse
from pathlib import Path
from typing import Any

import cv2
import torch
import numpy as np
from PIL import Image

from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output_keyboard,
)
from sam3.model_builder import build_sam3_video_predictor
from annotate.config.profile import OBJ, OBJ_ID, DATA_ROOT, VIDEO_CHUNK_SIZE, PROFILES, ProfileSpec
from annotate.utils.coco_io import save_outputs_merged_to_coco_json
from annotate.utils.sam3_util import *


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = parser.add_subparsers(dest="profile", required=True)

    for key, spec in PROFILES.items():
        sp = sub.add_parser(key)
        sp.add_argument("--video_path", type=str, default=DATA_ROOT/f"scenario_{spec.scenario}/{spec.date_dir}/videos/chunk-{spec.chunk:03d}/observation.images.cam{spec.cam}/episode_{spec.episode:06d}.mp4")
        sp.add_argument("--out_dir", type=str, default=DATA_ROOT/f"scenario_{spec.scenario}_cam_{spec.cam}_annotate/{spec.date_dir}")
        sp.add_argument(
            "--show_config", 
            action="store_true",
            help="Print derived paths (BASE/CHUNK_REL/OUT_PREFIX) and exit")
    return parser

class VideoPromptRunner:
    KEYS = ["out_obj_ids", "out_boxes_xywh", "out_binary_masks"]

    def __init__(self, predictor, args, spec: ProfileSpec, point_prompt: list=None):
        self.point_prompt = point_prompt
        self.predictor = predictor
        self.args = args
        self.spec = spec

    @staticmethod
    def load_video_frames_for_vis(video_path: str):
        video_frames_for_vis = []
        if isinstance(video_path, Path):
            video_path = str(video_path)
        if isinstance(video_path, str) and video_path.endswith(".mp4"):
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        return video_frames_for_vis
    
    def get_obj_by_id(self, obj_id):
        return next((o for o in self.point_prompt['objects'] if o["obj_id"] == obj_id), None)

    def propagate_in_video(self, session_id, obj_id):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            out = response["outputs"]
            out['out_obj_ids']=np.array([obj_id],dtype=out['out_obj_ids'].dtype)
            outputs_per_frame[response["frame_index"]] = out

        return outputs_per_frame

    def merge_outputs_per_frame(self, outputs_list):
        outputs_merged = {}
        KEYS = ["out_obj_ids", "out_boxes_xywh", "out_binary_masks"]
        for outputs in outputs_list:
            for frame_idx, out in outputs.items():
                obj_ids = out.get("out_obj_ids", None)
                masks = out.get("out_binary_masks", None)
                if obj_ids is None or masks is None:
                    continue
                if len(obj_ids) != masks.shape[0]:
                    continue
                
                if frame_idx not in outputs_merged:
                    outputs_merged[frame_idx]={k:out[k] for k in KEYS}
                else:
                    for k in KEYS:
                        if k in out and isinstance(out[k], np.ndarray):
                            outputs_merged[frame_idx][k] = np.concatenate(
                                [outputs_merged[frame_idx][k],out[k]],
                                axis=0
                            )
        return outputs_merged

    def init_prompt_and_propagate(self, prompt, session_id, obj_id, frame_index = 0):
        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=prompt,
                obj_id=obj_id,
            )
        )
        out = response["outputs"]

        outputs_per_frame = self.propagate_in_video(session_id, obj_id)
        return outputs_per_frame, out

    def run_prompts_and_merge(
            self, 
            video_path:list, 
            *,
            objs: list[dict[str, Any]],
        ):
        # Start a session
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]
        outputs_list = []

        for cfg in objs:
            print(f"####### Segment result of object {OBJ[cfg["obj_id"]]['name']} ######")
            
            point_obj = self.get_obj_by_id(cfg['obj_id'])
            frame_index = point_obj.get("frame_index", 0)
            points = point_obj['points']
            labels = point_obj['labels']

            if len(points) == 0:
                continue

            outputs_per_frame, out = self.init_prompt_and_propagate(
                prompt = cfg["prompt"],
                session_id=session_id, 
                frame_index=cfg.get("frame_index", 0),
                obj_id=cfg["obj_id"],
            )

            # # visualization
            # frame_index = cfg.get("frame_index", 0)
            # key = visualize_formatted_frame_output_keyboard(
            #     frame_index,
            #     video_path,
            #     outputs_list=[prepare_masks_for_visualization({frame_index: out})],
            #     titles=[f"Segment Object: {OBJ[cfg["obj_id"]]['name']}"],
            #     figsize=(6, 4),
            # )
            if out['frame_stats']['num_obj_tracked'] != 0:
                response = self.predictor.handle_request(
                        request=dict(
                            type="remove_object",
                            session_id=session_id,
                            obj_id=out['out_obj_ids'][0],
                            )
                        )
            
            IMG_WIDTH, IMG_HEIGHT = video_path[0].size
            points_tensor = torch.tensor(
                abs_to_rel_coords(points, IMG_WIDTH, IMG_HEIGHT, coord_type="point"),
                dtype=torch.float32,
            )

            points_labels_tensor = torch.tensor(labels, dtype=torch.int32)
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    points=points_tensor,
                    point_labels=points_labels_tensor,
                    obj_id=cfg["obj_id"],
                )
            )
            # out = response['outputs']
            # print(f"####### Segment result of object {OBJ[cfg["obj_id"]]['name']} ######")
            # frame_index = cfg.get("frame_index", 0)
            # key = visualize_formatted_frame_output_keyboard(
            #     frame_index,
            #     video_path,
            #     outputs_list=[prepare_masks_for_visualization({frame_index: out})],
            #     titles=[f"Segment Object: {OBJ[cfg["obj_id"]]['name']}"],
            #     figsize=(6, 4),
            # )

            outputs_per_frame = self.propagate_in_video(session_id, cfg["obj_id"])
            outputs_list.append(outputs_per_frame)

        # merge outputs for bottle, coaster and hand
        outputs_merged = self.merge_outputs_per_frame(outputs_list)

        _ = self.predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        return outputs_merged

    def run_in_chunks_and_merge(self, video_frames_for_vis, chunk_size = VIDEO_CHUNK_SIZE):
        outputs_all = {}
        n_frames = len(video_frames_for_vis)
        is_first_chunk = 1

        # 从当前 subcommand/profile 拿到要跑哪些对象
        ORDER = list(self.spec.objects)
        # e.g. ["bottle","pad","hand"] or ["bottle","box","hand"]

        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            chunk_frames = video_frames_for_vis[start:end]

            if is_first_chunk == 1:
                objs = []
                for name in ORDER:
                    obj_id = OBJ_ID[name]
                    prompt = self.spec.prompt(obj_id)
                    frame_index = self.spec.frame(obj_id)
                    if not prompt:   # 可选：允许某个对象不做
                        continue
                    objs.append(dict(
                        obj_id=obj_id,
                        prompt=prompt,
                        frame_index=frame_index,
                    ))

                chunk_out = self.run_prompts_and_merge(
                    video_path = chunk_frames,
                    objs=objs
                    )  # dict: {local_frame_idx: out}
                
                is_first_chunk = 0
            else:
                prev = outputs_all[start - 1]
                id2box = {
                    int(obj_id): box 
                    for obj_id, box in zip(prev["out_obj_ids"], prev["out_boxes_xywh"])}
                objs = []
                for name in ORDER:
                    obj_id = OBJ_ID[name]
                    prompt = getattr(self.args, f"{name}_prompt")
                    if not prompt:
                        continue

                    box = id2box.get(obj_id, None)
                    objs.append(dict(
                        obj_id=obj_id,
                        prompt=prompt,
                        bounding_boxes=[box] if box is not None else None,
                        bounding_box_labels=[1] if box is not None else None,
                        # frame_index 不传也行（init_prompt_and_propagate 默认 0）
                    ))
                chunk_out = self.run_prompts_and_merge(
                    video_path = chunk_frames,
                    objs = objs
                    )
                print("test")
            for local_idx, out in chunk_out.items():
                global_idx = start + local_idx
                outputs_all[global_idx] = out  # 把local帧号映射成全局帧号

        return outputs_all

def main():

    predictor = build_sam3_video_predictor()
    args = build_parser().parse_args()

    spec = PROFILES[args.profile]
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    refine_point_root = out_dir/"refine_folder"/"refine_point.json"

    with open(refine_point_root, "r") as f:
        point_prompt_list = json.load(f)

    for point_prompt in point_prompt_list:
        epi = point_prompt["episode"]

        video_path = Path(args.video_path).with_name(f"episode_{epi:06d}.mp4")
        out_coco_path = out_dir / 'refine_folder' / f"{video_path.stem}.coco.json"

        args.video_path = video_path
        runner = VideoPromptRunner(predictor, args, spec, point_prompt)
        video_frames = runner.load_video_frames_for_vis(video_path)
        video_frames_pil = [Image.fromarray(frame) for frame in video_frames]
        outputs_merged = runner.run_in_chunks_and_merge(
            video_frames_for_vis=video_frames_pil,
        )
        save_outputs_merged_to_coco_json(
            outputs_merged,
            out_coco_path,
            video_name=Path(video_path).stem
        )

if __name__ == "__main__":
    main()
