import os
import argparse
from pathlib import Path
from typing import Dict, Tuple
from typing import Any, Optional
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor
from coco_io import save_outputs_merged_to_coco_json


OBJ = {
    "hand":   {"id": 0, "prompt": "white-and-black robotic hand", "color_bgr": (255, 0, 0)},
    "bottle": {"id": 1, "prompt": "bottle",                      "color_bgr": (0, 255, 0)},
    "pad":    {"id": 2, "prompt": "blue or grey circle",         "color_bgr": (0, 0, 255)},
    "box":    {"id": 3, "prompt": "grey rectangular tray",       "color_bgr": (255, 255, 0)},
}
OBJ_ID   = {name: cfg["id"] for name, cfg in OBJ.items()}
OBJ_NAME = {cfg["id"]: name for name, cfg in OBJ.items()}
DEFAULT_PROMPT_BY_ID = {cfg["id"]: cfg["prompt"] for cfg in OBJ.values()}
DEFAULT_COLOR_BY_ID  = {cfg["id"]: cfg["color_bgr"] for cfg in OBJ.values()}

SCENARIO_OBJECTS = {
    1: ("hand", "bottle", "pad"),
    2: ("hand", "bottle", "box"),
}

VIDEO_CHUNK_SIZE = 2000

DATA_ROOT = Path("/home/grail/training_data/real_data")
OUT_TAG = "marked"

def make_video_path(
    scenario: int,
    cam: int,
    episode: int,
    chunk: int = 0,
    date_dir: Optional[str] = None,  # e.g. "2025.12.20" or None
) -> str:
    parts = [DATA_ROOT, f"scenario_{scenario}"]
    if date_dir:  # None / "" -> skip
        parts.append(date_dir)
    parts += [
        "videos",
        f"chunk-{chunk:03d}",
        f"observation.images.cam{cam}",
        f"episode_{episode:06d}.mp4",
    ]
    return str(Path(*parts))

def profile_key(scenario: int, cam: int) -> str:
    return f"s{scenario}c{cam}"

def make_out_dir(scenario: int, cam: int, tag: str) -> str:
    return str(DATA_ROOT / f"scenario_{scenario}_cam_{cam}_{tag}")

@dataclass(frozen=True)
class ProfileSpec:
    scenario: int
    cam: int
    episode: int
    out_tag: str = OUT_TAG
    date_dir: Optional[str] = None
    chunk: int = 0

    objects: Tuple[str, ...] = ("bottle", "pad", "hand")  # 默认给 scenario1 的集合

    prompts: Dict[int, str] = field(default_factory=dict)      # obj_id -> prompt
    init_frame: Dict[int, int] = field(default_factory=dict)   # obj_id -> frame index
    colors_bgr: Dict[int, Tuple[int,int,int]] = field(default_factory=dict)  # overrides
    
    @property
    def video_path(self) -> str:
        return make_video_path(
            scenario=self.scenario,
            cam=self.cam,
            episode=self.episode,
            chunk=self.chunk,
            date_dir=self.date_dir,
        )
    
    @property
    def out_dir(self) -> str:
        return make_out_dir(self.scenario, self.cam, self.out_tag)
    
    def prompt(self, obj_id: int) -> str:
        return self.prompts.get(obj_id, DEFAULT_PROMPT_BY_ID[obj_id])
    
    def frame(self, obj_id: int) -> int:
        return self.init_frame.get(obj_id, 0)

    def color(self, obj_id: int) -> Tuple[int,int,int]:
        return self.colors_bgr.get(obj_id, DEFAULT_COLOR_BY_ID[obj_id])

def make_profile(*, scenario: int, cam: int, **kwargs) -> Tuple[str, ProfileSpec]:
    if "objects" not in kwargs:
        kwargs["objects"] = SCENARIO_OBJECTS[scenario]  # 若 scenario 不在表里会 KeyError（更早暴露配置问题）
    key = profile_key(scenario, cam)
    spec = ProfileSpec(scenario=scenario, cam=cam, **kwargs)
    return key, spec

# ---- define per scenario/cam profiles ----
PROFILES: Dict[str, ProfileSpec] = dict([
    # scenario 1
    make_profile(
        scenario=1, cam=2, episode=3,
        date_dir="2025.12.01",
    ),
    make_profile(
        scenario=1, cam=3, episode=9,
        prompts={OBJ_ID["hand"]: "robotic hand with thumb"},
        init_frame={OBJ_ID["bottle"]: 60},
        date_dir="2025.12.01",
    ),

    # scenario 2
    make_profile(
        scenario=2, cam=2, episode=9,
        date_dir="2025.12.02",
    ),
    make_profile(
        scenario=2, cam=3, episode=9,
        date_dir="2025.12.02",
        prompts={OBJ_ID["hand"]: "robotic hand with thumb"},
        init_frame={OBJ_ID["bottle"]: 60},
    ),

    # scenario 3
])

def add_args_for_profile(p: argparse.ArgumentParser, spec: ProfileSpec) -> None:
    # 这两个是“最终值”（自动生成），但仍允许命令行覆盖
    p.add_argument("--video_path", type=str, default=spec.video_path)
    p.add_argument("--out_dir", type=str, default=spec.out_dir)
    p.add_argument("--show_config", action="store_true",
                    help="Print derived paths (BASE/CHUNK_REL/OUT_PREFIX) and exit")

    # 每个 object 的 prompt / frame / color
    for name, oid in OBJ_ID.items():
        p.add_argument(f"--{name}_prompt", type=str, default=spec.prompt(oid))
        p.add_argument(f"--frame_idx_{name}", type=int, default=spec.frame(oid))
        b, g, r = spec.color(oid)
        p.add_argument(f"--color_{name}", type=int, nargs=3, default=[b, g, r], metavar=("B","G","R"))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = parser.add_subparsers(dest="profile", required=True)

    for key, spec in PROFILES.items():
        sp = sub.add_parser(key)
        add_args_for_profile(sp, spec)

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
        is_hand = (obj_id == OBJ_ID["hand"])
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

class VideoMaskRenderer:
    def __init__(
        self,
        video_frames_for_vis:list,
        outputs_merged: dict,
        color_by_id: dict[int, tuple[int, int, int]],
        fps: int = 30,
        alpha: float = 0.18,
        thickness: int = 2,
        fourcc: str = "mp4v",
    ):
        self.video_frames_for_vis = video_frames_for_vis
        self.outputs_merged = outputs_merged
        self.color_by_id = color_by_id  # BGR colors
        self.alpha = float(alpha)
        self.thickness = int(thickness)
        self.fps = int(fps)
        self.fourcc = fourcc
        self.H, self.W = self.video_frames_for_vis[0].shape[:2]

    def overlay_mask_bgr(
        self,
        frame_bgr, 
        mask_bool, 
        fill_color_bgr, 
        outline_color_bgr=None, 
    ):
        """frame_bgr: (H,W,3) uint8, mask_bool: (H,W) bool"""
        if mask_bool is None or mask_bool.size == 0:
            return frame_bgr
        m = (mask_bool.astype(np.uint8) * 255)  # uint8, values 0/255
        if m.max() == 0:
            return frame_bgr
        
        if outline_color_bgr is None:
            outline_color_bgr = fill_color_bgr

        overlay = frame_bgr.copy()
        overlay[m>0] = fill_color_bgr
        frame_bgr = cv2.addWeighted(overlay, self.alpha, frame_bgr, 1 - self.alpha, 0)

        res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        cv2.drawContours(frame_bgr, contours, -1, outline_color_bgr, self.thickness)

        return frame_bgr

    def render(self, out_video_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(
            out_video_path, 
            fourcc, 
            self.fps, 
            (self.W, self.H),
        )
        assert vw.isOpened(), f"Failed to open VideoWriter: {out_video_path}"
        n_frames = len(self.video_frames_for_vis)
        for frame_idx in range(n_frames):
            # if frame_idx not in self.outputs_merged: 
            #     continue
            frame_bgr = cv2.cvtColor(self.video_frames_for_vis[frame_idx], cv2.COLOR_RGB2BGR) #original
            out = self.outputs_merged.get(frame_idx, None)
            if out is not None:
                masks = out.get("out_binary_masks", None)  # (3,H,W)
                obj_ids = out.get("out_obj_ids", None)
                if masks is not None and obj_ids is not None:
                    N = obj_ids.shape[0]
                    for i in range(N):
                        mask = masks[i]
                        obj_id = obj_ids[i]
                        if mask is None or mask.size == 0 or not mask.any():
                            continue
                        c = self.color_by_id.get(obj_id)
                        frame_bgr = self.overlay_mask_bgr(frame_bgr, mask, c)
            vw.write(frame_bgr)
        vw.release()
        print("Saved video to:", out_video_path)
        return

def main():
    predictor = build_sam3_video_predictor()
    args = build_parser().parse_args()
    spec = PROFILES[args.profile]

    if args.show_config:
        print(f'BASE="{DATA_ROOT}/scenario_{spec.scenario}"')
        print(f'CHUNK_REL="videos/chunk-{spec.chunk:03d}/observation.images.cam{spec.cam}"')
        print(f'OUT_PREFIX="{make_out_dir(spec.scenario, spec.cam, spec.out_tag)}"')
        return

    video_path = args.video_path
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    stem = Path(video_path).stem
    out_coco_path = Path(out_dir) / f"{stem}.coco.json"
    out_video_path = Path(out_dir) / f"{stem}_{spec.out_tag}.mp4"

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

    # test read coco and get outputs_merged
    # outputs_merged = load_outputs_merged_from_coco_json(out_coco_path)

    color_by_id = {
        OBJ_ID[name]: tuple(getattr(args, f"color_{name}"))
        for name in spec.objects
    }
    renderer = VideoMaskRenderer(
        video_frames_for_vis=video_frames,
        outputs_merged=outputs_merged,
        color_by_id=color_by_id,
        fps=30,
        alpha=0.18,
        thickness=2,
    )
    renderer.render(out_video_path)

if __name__ == "__main__":
    main()
