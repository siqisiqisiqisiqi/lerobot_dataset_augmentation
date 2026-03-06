import os
import argparse
from pathlib import Path
import numpy as np
from typing import Any
from PIL import Image
import cv2

from sam3.model_builder import build_sam3_video_predictor
from annotate.config.profile import OBJ_ID, DATA_ROOT, VIDEO_CHUNK_SIZE, PROFILES, ProfileSpec
from annotate.utils.coco_io import save_outputs_merged_to_coco_json

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

    def __init__(self, predictor, args, spec: ProfileSpec):
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

    def propagate_in_video(self, session_id, obj_id):
        # we will just propagate from frame 0 to the end of the video
        is_hand = (obj_id == 0)
        outputs_per_frame = {}
        if not is_hand:
            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                out = response["outputs"]
                out['out_obj_ids']=np.array([obj_id],dtype=out['out_obj_ids'].dtype)
                outputs_per_frame[response["frame_index"]] = out

        else:
            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                out = response["outputs"]
                # extract right hand masks
                binary_masks = out["out_binary_masks"]
                if binary_masks is None or len(binary_masks) == 0:
                    continue
                x_max = []
                for m in binary_masks:
                    xs = np.where(m)[1]
                    x_max.append(xs.max() if xs.size > 0 else -1)
                right_hand_idx = int(np.argmax(x_max))
                new_out = {}
                N=len(binary_masks)
                for k,v in out.items():
                    if isinstance(v,np.ndarray) and v.shape[0]==N:
                        new_out[k]=v[right_hand_idx:right_hand_idx+1]
                    else:
                        new_out[k]=v
                new_out['out_obj_ids']=np.array([obj_id],dtype=out['out_obj_ids'].dtype)
                outputs_per_frame[response["frame_index"]] = new_out

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

    def init_prompt_and_propagate(self,prompt, session_id, obj_id, bounding_boxes=None, bounding_box_labels=None, frame_index = 0):
        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=prompt,
                bounding_boxes=bounding_boxes,
                bounding_box_labels = bounding_box_labels
            )
        )
        outputs_per_frame = self.propagate_in_video(session_id, obj_id)
        return outputs_per_frame

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
            outputs_per_frame = self.init_prompt_and_propagate(
                prompt = cfg["prompt"],
                session_id=session_id, 
                bounding_boxes = cfg.get("bounding_boxes", None), 
                bounding_box_labels = cfg.get("bounding_box_labels", None), 
                frame_index=cfg.get("frame_index", 0),
                obj_id=cfg["obj_id"],
            )
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
                    prompt = getattr(self.args, f"{name}_prompt")
                    frame_index = getattr(self.args, f"frame_idx_{name}")
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

    BASE=DATA_ROOT/f"scenario_{spec.scenario}"
    CHUNK_REL=f"videos/chunk-{spec.chunk:03d}/observation.images.cam{spec.cam}"
    VIDEO_PATH = f"episode_{spec.episode:06d}.mp4"
    OUT_PREFIX=DATA_ROOT/f"scenario_{spec.scenario}_cam_{spec.cam}_annotate"

    # use .sh script
    if args.show_config:
        print(f'BASE="{BASE}"')
        print(f'CHUNK_REL="{CHUNK_REL}"')
        print(f"VIDEO_PATH={VIDEO_PATH}")
        print(f'OUT_PREFIX="{OUT_PREFIX}"')
        return
    # END use .sh script

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    video_path = Path(args.video_path)
    out_coco_path = out_dir / f"{video_path.stem}.coco.json"

    runner = VideoPromptRunner(predictor, args, spec)
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
